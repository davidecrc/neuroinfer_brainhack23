 function updateLUT() {
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

        colors.forEach((hsl) => colors.push(hslToRgb(hsl));

        const lutData = [];
        lutData.push([parseInt(minvalue.value), colors[3]]);
        lutData.push([1, colors[4]]);
        lutData.push([parseInt(maxvalue.value), colors[5]]);

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

  return [round(r * 255), round(g * 255), round(b * 255)];
}