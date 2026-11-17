from acprof.host.metadata import machine


def test_preparation_topology_deduplicates_shared_cache(tmp_path):
    cpu_root, node_root = tmp_path / 'cpu', tmp_path / 'node'
    cpu_root.mkdir()
    (cpu_root / 'online').write_text('0-1\n')
    for cpu in (0, 1):
        topology = cpu_root / f'cpu{cpu}' / 'topology'
        topology.mkdir(parents=True)
        (topology / 'physical_package_id').write_text('0\n')
        (topology / 'core_id').write_text('0\n')
        cache = cpu_root / f'cpu{cpu}' / 'cache' / 'index0'
        cache.mkdir(parents=True)
        for name, value in {'level': '1', 'type': 'Data', 'size': '32K', 'shared_cpu_list': '0-1'}.items():
            (cache / name).write_text(value)
    node = node_root / 'node0'
    node.mkdir(parents=True)
    (node / 'cpulist').write_text('0-1\n')
    result = machine.cpu_topology_metadata(cpu_root=cpu_root, node_root=node_root)
    assert result['logical_cpu_count'] == 2
    assert result['physical_core_count'] == 1
    assert result['online_cpus'] == [0, 1]
    assert result['caches'] == [{'level': 1, 'type': 'Data', 'size_bytes': 32768, 'shared_cpus': [0, 1]}]
    assert result['numa_nodes'] == [{'node': 0, 'cpus': [0, 1]}]


def test_incomplete_topology_does_not_fabricate_core_count(tmp_path):
    (tmp_path / 'cpu0' / 'topology').mkdir(parents=True)
    result = machine.cpu_topology_metadata(cpu_root=tmp_path, node_root=tmp_path / 'missing')
    assert result['physical_core_count'] is None
    assert result['online_cpus'] is None
    assert result['errors']
