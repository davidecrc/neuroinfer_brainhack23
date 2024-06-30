 window.updateLUT = function () {
        let slider1 = document.getElementById('colorRange1');
        let slider2 = document.getElementById('colorRange2');
        let slider3 = document.getElementById('colorRange3');
        let minvalue = document.getElementById('min');
        let maxvalue = document.getElementById('max');

        const colors = [
            parseInt(slider1.value),
            parseInt(slider2.value),
            parseInt(slider3.value)
        ];
        console.log(colors);

        let lutData = [];
        step_grad = 50;

        for (let i = 0; i < step_grad; i++){
            color_i = colors[0] + (colors[1] - colors[0])*i/step_grad;
            rgbColors = hslToRgb(color_i/360, 1, 0.5);
            target_value = (parseInt(minvalue.value) + (1-parseInt(minvalue.value))*i/step_grad)/max_value;
            lutData.push([target_value, rgbColors[0], rgbColors[1], rgbColors[2]]);
        }

        for (let i = 0; i < step_grad; i++){
            color_i = colors[1] + (colors[2] - colors[1])*i/step_grad;
            rgbColors = hslToRgb(color_i/360, 1, 0.5);
            target_value = (1 + (parseInt(maxvalue.value)-1)*i/step_grad)/max_value;
            lutData.push([target_value, rgbColors[0], rgbColors[1], rgbColors[2]]);
        }

        for (let i = 0; i < step_grad; i++){
            color_i = colors[2];
            rgbColors = hslToRgb(color_i/360, 1, 0.5);
            target_value = (parseInt(maxvalue.value)/max_value + (1-parseInt(maxvalue.value)/max_value)*i/step_grad);
            lutData.push([target_value, rgbColors[0], rgbColors[1], rgbColors[2]]);
        }

        console.log(lutData);

        let lut_id = (Math.random() + 1).toString(36).substring(7); // random name to avoid cached files

        params["luts"] = [{"name": lut_id, "data": lutData}];
        overlays.forEach( (overlay_id) =>
            params[getFileName(overlay_id)] = {lut: lut_id})

        params[getFileName(current_overlay)] = {lut: lut_id};
        update_papaya_viewer(current_overlay);
    }

function getFileName(path) {
    return path.split('/').pop();
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

  return [r, g, b];
}

function hueToRgb(p, q, t) {
  if (t < 0) t += 1;
  if (t > 1) t -= 1;
  if (t < 1/6) return p + (q - p) * 6 * t;
  if (t < 1/2) return q;
  if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
  return p;
}