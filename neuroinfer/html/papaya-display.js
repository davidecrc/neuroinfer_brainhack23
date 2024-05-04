// Function to display a plot in the Papaya viewer
window.displayPlot = function(imageData) {
    // Updating the Papaya viewer with the new data
    update_papaya_viewer(true);

    // Getting the graphics container element
    var graphicsContainer = document.getElementById("graphics-container");

    // Checking if the graphics container element exists
        // Creating an image element
        var img = new Image();

        // Setting the source of the image to the base64-encoded data
        img.src = 'data:image/png;base64,' + imageData;

        // Setting max-width and max-height properties to fit the container
        img.style.maxWidth = '100%';
        img.style.maxHeight = '100%';

        // Appending the image to the graphics container
        graphicsContainer.innerHTML = '';
        graphicsContainer.appendChild(img);

};
