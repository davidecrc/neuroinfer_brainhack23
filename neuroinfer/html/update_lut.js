 window.updateLUT = function () {
        const slider1 = document.getElementById('colorRange1');
        const slider2 = document.getElementById('colorRange2');
        const slider3 = document.getElementById('colorRange3');
        const minvalue = document.getElementById('min');
        const maxvalue = document.getElementById('max');

        const colors = [
            parseInt(slider1.value),
            parseInt(slider2.value),
            parseInt(slider3.value)
        ];
        console.log(colors);

        const rgbColors = [];

        // Iterate over each HSL value in the original array
        colors.forEach((hsl) => {
            // Convert HSL to RGB and push the result into the new array
            rgbColors.push(hslToRgb(hsl/360, 1, 0.5));
        });
        console.log(rgbColors);

        const lutData = [];
        lutData.push([parseInt(minvalue.value), rgbColors[0]]);
        lutData.push([1, rgbColors[1]]);
        lutData.push([parseInt(maxvalue.value), rgbColors[2]]);

        params["luts"] = [{"name": "Custom", "data": lutData}];
        for (let overlay_id in overlays) {
            params[overlay_id] = {lut: "Custom"};
        }
        params[current_overlay] = {lut: "Custom"};
        update_papaya_viewer(current_overlay);
    }


function hslToRgb(h, s, l) {
  let r, g, b;

  if (s === 0) {
    r = g = b = l; // achromatic
  } else {
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hueToRgb(p, q, h + 1/3);
    g = hueToRgb(p, q, h);
    b = hueToRgb(p, q, h - 1/3);
  }

  return [Math.floor(r * 255), Math.floor(g * 255), Math.floor(b * 255)];
}

function hueToRgb(p, q, t) {
  if (t < 0) t += 1;
  if (t > 1) t -= 1;
  if (t < 1/6) return p + (q - p) * 6 * t;
  if (t < 1/2) return q;
  if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
  return p;
}