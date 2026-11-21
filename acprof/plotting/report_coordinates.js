/* Exact measurement coordinates shared by CSV import and window statistics. */
"use strict";
window.ACProfCoordinates = (() => {
  function decimalCoordinate(value) {
    // Normalize coefficient and exponent separately; a repeat such as 1e10000 never expands into 10001 digits.
    const match = /^([+-]?)(\d*)(?:\.(\d*))?(?:e([+-]?)(\d+))?$/i.exec(value);
    if (!match || !(match[2] || match[3])) return null;
    const fraction = match[3] || "", digits = (match[2] + fraction).replace(/^0+/, "");
    const exponent = (match[5] || "").replace(/^0+/, "") || "0";
    // The project's Decimal context uses a signed, bounded exponent.
    if (exponent.length > 19) return null;
    let power = BigInt((match[4] || "") + exponent) - BigInt(fraction.length);
    if (power < -1999999999999999997n || power + BigInt(Math.max(0, digits.length - 1)) > 999999999999999999n) return null;
    if (!digits) return {key:"0", integer:true, coefficient:"0", power:0n};
    if (match[1] === "-") return null;
    let end = digits.length;
    while (digits[end - 1] === "0") end--;
    power += BigInt(digits.length - end);
    const coefficient = digits.slice(0, end), adjusted = power + BigInt(end - 1);
    let key = coefficient;
    // Match decimal_key_text for evidence shared with Python-generated bundles.
    if (power > 0n || adjusted < -6n) {
      const mantissa = end === 1 ? coefficient : coefficient[0] + "." + coefficient.slice(1);
      key = mantissa + "E" + (adjusted > 0n ? "+" : "") + adjusted;
    } else if (power < 0n) {
      const point = Number(adjusted) + 1;
      key = point <= 0 ? "0." + "0".repeat(-point) + coefficient :
        coefficient.slice(0, point) + "." + coefficient.slice(point);
    }
    return {key, integer:power >= 0n, coefficient, power};
  }

  function repeatIndex(value) {
    if (typeof value === "number") return Number.isSafeInteger(value) && value >= 0 ? value : null;
    if (typeof value !== "string" || Number.isSafeInteger(Number(value))) return null;
    const coordinate = decimalCoordinate(value);
    return coordinate?.integer && coordinate.key === value ? coordinate : null;
  }

  function ascending(left, right) {
    const a = left.coefficient, b = right.coefficient;
    const widthA = BigInt(a.length) + left.power, widthB = BigInt(b.length) + right.power;
    // A normalized coefficient ends in a nonzero digit, so a longer matching prefix sorts last.
    return widthA === widthB ? a < b : widthA < widthB;
  }

  function adjacent(left, right) {
    const a = left.coefficient, b = right.coefficient;
    if (left.power > 0n) {
      // Adding one to a value with trailing zeros leaves its coefficient prefix intact.
      return right.power === 0n && BigInt(b.length - a.length) === left.power &&
        b.startsWith(a) && b.endsWith("1") && /^0*$/.test(b.slice(a.length, -1));
    }
    let end = a.length;
    while (end && a[end - 1] === "9") end--;
    // Every carried nine becomes a trailing zero in the normalized successor.
    if (right.power !== BigInt(a.length - end)) return false;
    if (!end) return b === "1";
    return b.length === end && b.startsWith(a.slice(0, end - 1)) &&
      b.charCodeAt(end - 1) === a.charCodeAt(end - 1) + 1;
  }

  function repeatSequence(indices) {
    let previous = null, consecutive = true;
    for (const value of indices) {
      const current = repeatIndex(value);
      if (current === null) return "invalid";
      if (previous !== null) {
        if (typeof previous === "number" && typeof current === "number") {
          if (previous >= current) return "invalid";
          consecutive &&= previous + 1 === current;
        } else {
          const a = typeof previous === "number" ? decimalCoordinate(String(previous)) : previous;
          const b = typeof current === "number" ? decimalCoordinate(String(current)) : current;
          if (!ascending(a, b)) return "invalid";
          consecutive &&= adjacent(a, b);
        }
      }
      previous = current;
    }
    return consecutive ? "consecutive" : "nonconsecutive";
  }
  return {decimalCoordinate, repeatSequence};
})();
