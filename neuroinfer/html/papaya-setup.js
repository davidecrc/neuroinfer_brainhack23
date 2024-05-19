// Variable to store parameters for Papaya viewer
var params = {};
// Declare filenames globally
var filenames = [];
var words = [];
var overlays = [];
var currentCoords = [0, 0, 0];

// Event listener for when the DOM content is loaded
document.addEventListener("DOMContentLoaded", function() {
    // Setting default parameters for the Papaya viewer
    params["worldSpace"] = true;
    params["showControls"] = false;
    params["showControlBar"] = true;
    params["showOrientation"] = true;
    params["combineParametric"] = true;
    params["showImageButtons"] = true;
    params["mainView"] = 'axial';
    params["kioskMode"] = true;
    params["coordinate"] = currentCoords;

    // Initializing the Papaya viewer with the specified parameters
    update_papaya_viewer();
});
