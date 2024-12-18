// Variable to store parameters for Papaya viewer
var params = {};
// Declare filenames globally
var filenames = [];
var filenames_sys = [];
var words = [];
var overlays = [];
var currentCoords = [0, 0, 0];
var init_loader = 1;
var current_overlay = null;
var max_value = null;
var selected_atlas = null;
visualizer = document.getElementById("visualizer");
visualizer.setAttribute("class", "papaya");
visualizer.setAttribute("data-params", "params");
papaya.Container.startPapaya();

// Event listener for when the DOM content is loaded
document.addEventListener("DOMContentLoaded", function () {
  // Setting default parameters for the Papaya viewer
  params["worldSpace"] = true;
  // params["showControls"] = false;
  // params["showControlBar"] = true;
  // params["showOrientation"] = true;
  // params["combineParametric"] = true;
  // params["showImageButtons"] = true;
  // params["mainView"] = "axial";
  // params["kioskMode"] = true;
  params["coordinate"] = currentCoords;

  // Initializing the Papaya viewer with the specified parameters
  update_papaya_viewer();
});
