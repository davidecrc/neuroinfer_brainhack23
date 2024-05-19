document.addEventListener("DOMContentLoaded", function () {
    // Initialize Papaya Viewer
    update_papaya_viewer(filenames[0]);

    const minSlider = document.getElementById('minSlider');
    const midSlider = document.getElementById('midSlider');
    const maxSlider = document.getElementById('maxSlider');

    // Function to interpolate between two colors
    function interpolateColor(color1, color2, factor) {
        const result = color1.slice();
        for (let i = 0; i < 3; i++) {
            result[i] = Math.round(result[i] + factor * (color2[i] - color1[i]));
        }
        return result;
    }

    // Function to update LUT
    function updateLUT() {
        const minValue = parseInt(minSlider.value);
        const midValue = parseInt(midSlider.value);
        const maxValue = parseInt(maxSlider.value);

        const blue = [0, 0, 255];
        const green = [0, 255, 0];
        const red = [255, 0, 0];
        const white = [255, 255, 255];

        const lutData = [];

        // Interpolate colors for the gradient
        for (let i = 0; i < 256; i++) {
            let color;
            if (i <= minValue) {
                color = blue;
            } else if (i <= midValue) {
                const factor = (i - minValue) / (midValue - minValue);
                color = interpolateColor(blue, green, factor);
            } else if (i <= maxValue) {
                const factor = (i - midValue) / (maxValue - midValue);
                color = interpolateColor(green, red, factor);
            } else {
                color = white;
            }
            lutData.push([i, ...color]);
        }

        // Assuming viewer[0] is the target Papaya viewer
        const viewer = papayaContainers[0].viewer;

        // Apply the new LUT to the viewer's screen volume
        viewer.screenVolumes[0].lut = new papaya.viewer.ColorTable("Custom", lutData);
        viewer.screenVolumes[0].changeColorTable("Custom");

        // Redraw the viewer
        viewer.drawViewer(true);
    }

    // Add event listeners to sliders
    minSlider.addEventListener('input', updateLUT);
    midSlider.addEventListener('input', updateLUT);
    maxSlider.addEventListener('input', updateLUT);

    // Initial LUT setup
    updateLUT();
});