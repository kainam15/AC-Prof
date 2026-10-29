"""Independent Native Linux egress acceptance; does not enable a runtime policy.

Uses a pre-existing AC-Prof NLP service image and cached Model Store entry.
Creates only labelled temporary containers/networks; no downloads or host rules.
Requires Docker, passwordless sudo for the namespace helper/tcpdump, and an idle
measurement lock. Run with uv run --locked python scripts/validate_measurement_egress.py.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import platform
import pwd
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

from acprof.host.model_store.store import acquire_mount, store_root
from acprof.host.run_state import MeasurementLock
from acprof.monitors.energy_nvml import GPUEnergyMonitor

WAIT_AND_SERVE = """import os,time
from pathlib import Path
while not Path('/egress-gate/release').exists():
    time.sleep(0.02)
os.execvp('python',['python','-m','acprof.container.server'])
"""
PROBE = r'''import json, socket, sys
result={}
for name,host,port,kind in json.loads(sys.argv[1]):
    family=socket.AF_INET6 if ':' in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM if kind=='tcp' else socket.SOCK_DGRAM) as s:
        s.settimeout(0.8)
        try:
            s.connect((host,port))
            if kind!='tcp':
                payload = b'\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01' if kind=='dns' else b'acprof-egress-validation'
                s.send(payload); s.recv(512)
            result[name]={'connected':True}
        except OSError as exc:
            result[name]={'connected':False,'error':str(exc)}
print(json.dumps(result))
'''
DUMMY = "from http.server import HTTPServer,SimpleHTTPRequestHandler; HTTPServer(('0.0.0.0',8002),SimpleHTTPRequestHandler).serve_forever()"


class Canary(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'ok')

    def log_message(self, *_args):
        pass


class Experiment:
    def __init__(self, output: Path, image: str, gpu_uuid: str):
        self.output = output
        self.image = image
        self.gpu_uuid = gpu_uuid
        self.identifier = 'acprof-review-egress-' + uuid.uuid4().hex[:12]
        self.containers = []
        self.networks = []
        self.cidfiles = []
        self.commands = []
        self.helper = Path(__file__).with_name('egress_namespace.py').resolve()
        self.results = {'schema_version': 1, 'purpose': 'independent isolation acceptance',
                        'platform': platform.platform(), 'python': sys.version, 'cases': {}}
        self.client = requests.Session()
        self.client.trust_env = False

    def run(self, command, *, timeout=60):
        self.commands.append(command)
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'{command[:3]} exited {result.returncode}: {result.stderr[-4000:]}')
        return result.stdout.strip()

    def create(self, name, network, arguments, code):
        cidfile = self.output / (name + '.cid')
        self.cidfiles.append(cidfile)
        cid = self.run(['docker', 'create', '--cidfile', str(cidfile), '--name', self.identifier + '-' + name,
                        '--label', 'acprof.review-egress=' + self.identifier,
                        '--network', network, '--cap-drop=ALL', '--security-opt=no-new-privileges',
                        '--cpus=2', '--memory=2g', '-p', '127.0.0.1::8002',
                        *arguments, '--entrypoint', 'python', self.image, '-c', code])
        assert re.fullmatch('[0-9a-f]{64}', cid), cid
        self.containers.append(cid)
        self.run(['docker', 'start', cid])
        info = json.loads(self.run(['docker', 'inspect', cid]))[0]
        return cid, info

    def network(self, name, *arguments):
        network = self.run(['docker', 'network', 'create', '--label', 'acprof.review-egress=' + self.identifier,
                            *arguments, self.identifier + '-' + name])
        self.networks.append(network)
        return network, json.loads(self.run(['docker', 'network', 'inspect', network]))[0]

    def probe(self, cid, targets):
        return json.loads(self.run(['docker', 'exec', cid, 'python', '-c', PROBE, json.dumps(targets)], timeout=15))

    def policy(self, info, *, inspect=False):
        pid = info['State']['Pid']
        ticks = int(Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[19])
        command = ['sudo', '-n', '/usr/bin/python3', str(self.helper), str(pid), str(ticks)]
        if inspect:
            command.append('--inspect')
        return json.loads(self.run(command))

    def wait_ready(self, url, *, path='/ready', timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result = self.client.get(url + path, timeout=1)
                if result.status_code == 200:
                    return result
            except requests.RequestException:
                pass
            time.sleep(0.1)
        raise RuntimeError('service did not become ready: ' + url)

    def remove(self, cid):
        self.run(['docker', 'rm', '-f', cid])
        self.containers.remove(cid)

    def internal_candidate(self):
        network, info = self.network('internal', '--internal')
        gateway = info['IPAM']['Config'][0]['Gateway']
        cid, container = self.create('internal', network, [], DUMMY)
        ip = next(iter(container['NetworkSettings']['Networks'].values()))['IPAddress']
        self.wait_ready('http://' + ip + ':8002', path='/')
        listener = ThreadingHTTPServer((gateway, 0), Canary)
        thread = threading.Thread(target=listener.serve_forever, daemon=True)
        thread.start()
        try:
            outbound = self.probe(cid, [('host_service', gateway, listener.server_port, 'tcp'), ('public', '1.1.1.1', 443, 'tcp')])
        finally:
            listener.shutdown()
            listener.server_close()
            thread.join()
        self.results['cases']['internal_bridge'] = {
            'published_bindings': container['NetworkSettings']['Ports'], 'direct_host_http': True,
            'egress': outbound, 'accepted': False,
            'reason': 'host services remain reachable; current loopback publication is not preserved',
        }
        assert outbound['host_service']['connected'], outbound
        self.remove(cid)
        print('internal bridge: host service reachable; candidate rejected', flush=True)

    def extra_protocols(self, *, ipv6=False):
        if ipv6:
            try:
                with socket.socket(socket.AF_INET6, socket.SOCK_STREAM):
                    pass
            except OSError as exc:
                if exc.errno not in {errno.EAFNOSUPPORT, errno.EPROTONOSUPPORT}:
                    raise
                self.results['cases']['ipv6_protocols'] = {'status': 'unavailable', 'reason': str(exc)}
                print('IPv6 unavailable in this kernel; no IPv6 acceptance claim', flush=True)
                return
        name = 'ipv6' if ipv6 else 'ipv4'
        subnet = f"fd42:{uuid.uuid4().hex[:4]}:{uuid.uuid4().hex[:4]}::/64"
        network, network_info = self.network(name, *(['--ipv6', '--subnet', subnet] if ipv6 else []))
        gateway4 = next(item['Gateway'] for item in network_info['IPAM']['Config'] if ':' not in item['Subnet'])
        gateway = next(item['Gateway'] for item in network_info['IPAM']['Config'] if (':' in item['Subnet']) == ipv6)
        dual_server = "import socket; from http.server import HTTPServer,SimpleHTTPRequestHandler; HTTPServer.address_family=socket.AF_INET6; HTTPServer(('::',8002),SimpleHTTPRequestHandler).serve_forever()"
        cid, info = self.create(name + '-protocols', network, [], dual_server if ipv6 else DUMMY)
        peer, peer_info = self.create(name + '-protocols-peer', network, [], dual_server if ipv6 else DUMMY)
        field = 'GlobalIPv6Address' if ipv6 else 'IPAddress'
        target = next(iter(info['NetworkSettings']['Networks'].values()))[field]
        peer_ip = next(iter(peer_info['NetworkSettings']['Networks'].values()))[field]
        target_url = f'http://[{target}]:8002' if ipv6 else f'http://{target}:8002'
        peer_url = f'http://[{peer_ip}]:8002' if ipv6 else f'http://{peer_ip}:8002'
        self.wait_ready(peer_url, path='/')
        self.wait_ready(target_url, path='/')

        class IPv6Server(ThreadingHTTPServer):
            address_family = socket.AF_INET6

        listener = (IPv6Server if ipv6 else ThreadingHTTPServer)((gateway, 0), Canary)
        thread = threading.Thread(target=listener.serve_forever, daemon=True)
        thread.start()
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.bind((gateway4, 0))
        udp.settimeout(0.1)
        stopped = threading.Event()

        def echo():
            while not stopped.is_set():
                try:
                    data, address = udp.recvfrom(512)
                    udp.sendto(data, address)
                except TimeoutError:
                    continue

        echo_thread = threading.Thread(target=echo, daemon=True)
        echo_thread.start()
        try:
            targets = [('host_service', gateway, listener.server_port, 'tcp'), ('peer_service', peer_ip, 8002, 'tcp'),
                       ('udp_host', gateway4, udp.getsockname()[1], 'udp'), ('docker_dns', '127.0.0.11', 53, 'dns')]
            before = self.probe(cid, targets)
            for check in ('host_service', 'peer_service', 'udp_host'):
                assert before[check]['connected'], before
            self.policy(info)
            after = self.probe(cid, targets)
            assert all(not item['connected'] for item in after.values()), after
            response = self.wait_ready(target_url, path='/')
            assert response.status_code == 200
            self.results['cases'][name + '_protocols'] = {'status': 'verified', 'baseline': before, 'controlled': after,
                                                       'host_http_response': True, 'rules_after': self.policy(info, inspect=True)}
        finally:
            listener.shutdown()
            listener.server_close()
            thread.join()
            stopped.set()
            echo_thread.join(timeout=2)
            udp.close()
        self.remove(peer)
        self.remove(cid)
        print(name + ', UDP and embedded DNS: egress blocked; host response preserved', flush=True)

    def capture_requests(self, name, cid, info, url, *, gpu):
        ip = info['NetworkSettings']['Networks']['bridge']['IPAddress']
        path = self.output / (name + '.pcap')
        stderr_path = self.output / (name + '-tcpdump.stderr')
        command = ['sudo', '-n', 'tcpdump', '-n', '-U', '-s', '0', '-i', 'docker0', '-Z', pwd.getpwuid(os.getuid()).pw_name,
                   '-w', str(path), 'tcp port 8002 and host ' + ip]
        self.commands.append(command)
        responses = []
        monitor = GPUEnergyMonitor(device_uuid=self.gpu_uuid, idle_seconds=0) if gpu else None
        with stderr_path.open('w') as diagnostics:
            capture = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=diagnostics)
            try:
                deadline = time.monotonic() + 5
                while 'listening on' not in stderr_path.read_text():
                    if capture.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(stderr_path.read_text())
                    time.sleep(0.05)
                if monitor:
                    monitor.start()
                for index in range(12):
                    started = time.perf_counter()
                    response = self.client.post(url + '/predict', json={'text': 'The capital of France is [MASK].', 'batch_size': 1, 'params': {}},
                                                headers={'X-Req-Id': f'{name}:{index}', 'Connection': 'close'}, timeout=30)
                    response.raise_for_status()
                    responses.append({'request': index, 'latency_app_s': time.perf_counter() - started, 'status': response.status_code})
                if monitor:
                    monitor.stop()
                # Let libpcap drain after the target window has ended.
                time.sleep(1.1)
            finally:
                if monitor:
                    monitor.close()
                if capture.poll() is None:
                    self.run(['sudo', '-n', 'kill', '-INT', str(capture.pid)])
                try:
                    capture.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.run(['sudo', '-n', 'kill', '-KILL', str(capture.pid)])
                    capture.wait()
        current = json.loads(self.run([sys.executable, '-m', 'acprof.packet.sniff_parse_pcap', str(path), '8002']))
        old_path = self.output / 'legacy_parser.py'
        legacy = json.loads(self.run([sys.executable, str(old_path), str(path), '8002']))
        assert current == legacy, 'packet metrics differ from three-pass parser'
        assert len(current['requests']) == len(responses) == 12
        assert all(row['wire_bytes_status'] == 'available' for row in current['requests'].values())
        (self.output / (name + '-packets.json')).write_text(json.dumps(current, indent=2) + '\n')
        result = {'responses': responses, 'pcap_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'legacy_packet_equality': 'exact', 'parsed_requests': 12, 'capture_interface': 'docker0'}
        if monitor:
            result['gpu_telemetry'] = monitor.telemetry_snapshot()
        return result

    def model_case(self, name, mount, *, gpu, controlled):
        arguments = [*mount.args, '-e', 'TASK_TYPE=fill-mask', '-e', 'TASK_FAMILY=nlp',
                     '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'HF_HUB_DISABLE_TELEMETRY=1',
                     '-e', f'USE_GPU={int(gpu)}', '-e', 'ACPROF_REQUEST_TIMEOUT_S=30']
        if gpu:
            arguments += ['--gpus', 'device=' + self.gpu_uuid, '-e', 'CUDA_VISIBLE_DEVICES=0']
        gate = self.output / (name + '-gate')
        gate.mkdir()
        arguments += ['--mount', f'type=bind,src={gate},dst=/egress-gate,readonly']
        started = time.perf_counter()
        cid, info = self.create(name, 'bridge', arguments, WAIT_AND_SERVE)
        if controlled:
            policy = self.policy(info)
        else:
            policy = None
        (gate / 'release').touch()
        port = info['NetworkSettings']['Ports']['8002/tcp'][0]['HostPort']
        url = 'http://127.0.0.1:' + port
        ready = self.wait_ready(url).json()
        assert str(ready['device']).startswith('cuda') if gpu else ready['device'] == 'cpu', ready
        startup = time.perf_counter() - started
        gateway = info['NetworkSettings']['Networks']['bridge']['Gateway']
        listener = ThreadingHTTPServer((gateway, 0), Canary)
        thread = threading.Thread(target=listener.serve_forever, daemon=True)
        thread.start()
        try:
            peer, peer_info = self.create(name + '-peer', 'bridge', [], DUMMY)
            peer_ip = peer_info['NetworkSettings']['Networks']['bridge']['IPAddress']
            self.wait_ready('http://' + peer_ip + ':8002', path='/')
            targets = [('host_service', gateway, listener.server_port, 'tcp'), ('peer_lan', peer_ip, 8002, 'tcp'), ('public', '1.1.1.1', 443, 'tcp')]
            outbound = self.probe(cid, targets)
            if controlled:
                assert all(not item['connected'] for item in outbound.values()), outbound
            else:
                assert outbound['host_service']['connected'], outbound
                assert outbound['peer_lan']['connected'], outbound
            self.remove(peer)
        finally:
            listener.shutdown()
            listener.server_close()
            thread.join()
        # No probe servers or extra queries run during these target requests.
        measured = self.capture_requests(name, cid, info, url, gpu=gpu)
        result = {'device': 'gpu' if gpu else 'cpu', 'controlled': controlled, 'ready': ready,
                  'startup_elapsed_s': startup, 'startup_semantics': 'includes create/start, gate, policy and ready wait',
                  'egress': outbound, **measured}
        if controlled:
            result['rules_before'] = policy
            result['rules_after'] = self.policy(info, inspect=True)
        self.results['cases'][name] = result
        (self.output / (name + '-container.log')).write_text(self.run(['docker', 'logs', cid]))
        self.remove(cid)
        print(name + ': ready/predict passed; 12 requests; legacy packet fields identical', flush=True)

    def close(self):
        failures = []
        ids = set(self.containers)
        for path in self.cidfiles:
            if path.exists() and re.fullmatch('[0-9a-f]{64}', path.read_text().strip()):
                ids.add(path.read_text().strip())
        for cid in ids:
            check = subprocess.run(['docker', 'inspect', cid], capture_output=True, text=True)
            if check.returncode:
                continue
            info = json.loads(check.stdout)[0]
            if info['Config']['Labels'].get('acprof.review-egress') != self.identifier:
                failures.append('ownership mismatch: ' + cid)
                continue
            logs = subprocess.run(['docker', 'logs', cid], capture_output=True, text=True)
            (self.output / (cid[:12] + '-cleanup.log')).write_text(logs.stdout + logs.stderr)
            result = subprocess.run(['docker', 'rm', '-f', cid], capture_output=True, text=True)
            if result.returncode:
                failures.append(result.stderr)
        for network in reversed(self.networks):
            result = subprocess.run(['docker', 'network', 'rm', network], capture_output=True, text=True)
            if result.returncode:
                failures.append(result.stderr)
        self.client.close()
        self.results['cleanup_errors'] = failures
        (self.output / 'commands.json').write_text(json.dumps(self.commands, indent=2) + '\n')
        (self.output / 'results.json').write_text(json.dumps(self.results, indent=2) + '\n')
        if failures:
            raise RuntimeError('cleanup incomplete: ' + '; '.join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='Existing tiny BERT AC-Prof service image')
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocols-only', action='store_true', help='Run IPv4/IPv6 TCP, UDP and DNS checks only')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    with MeasurementLock():
        experiment = Experiment(args.output, args.image, args.gpu_uuid)
        mount = None
        try:
            experiment.image = experiment.run(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'])
            experiment.results['image_id'] = experiment.image
            experiment.results['source_head'] = experiment.run(['git', 'rev-parse', 'HEAD'])
            experiment.results['validation_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            experiment.results['policy_helper_sha256'] = hashlib.sha256(experiment.helper.read_bytes()).hexdigest()
            experiment.results['docker_version'] = experiment.run(['docker', 'version', '--format', '{{.Server.Version}}'])
            experiment.results['nft_version'] = experiment.run(['nft', '--version'])
            experiment.results['gpu_uuid'] = args.gpu_uuid
            manifest = json.loads(experiment.run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'cat', experiment.image, '/app/runtime_environment.json']))
            experiment.results['model_revision'] = manifest.get('model_snapshot_revision')
            (args.output / 'legacy_parser.py').write_text(experiment.run(['git', 'show', 'acc6345:acprof/packet/sniff_parse_pcap.py']) + '\n')
            if manifest.get('model_store'):
                manifest['model_store']['host_path'] = str(store_root())
            mount = acquire_mount(manifest)
            if not args.protocols_only:
                experiment.internal_candidate()
                for gpu in [False, True]:
                    for controlled in [False, True]:
                        name = ('gpu' if gpu else 'cpu') + ('-controlled' if controlled else '-baseline')
                        experiment.model_case(name, mount, gpu=gpu, controlled=controlled)
            experiment.extra_protocols()
            experiment.extra_protocols(ipv6=True)
        except BaseException as exc:
            experiment.results["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            try:
                experiment.close()
            finally:
                if mount is not None:
                    mount.close()


if __name__ == '__main__':
    main()
