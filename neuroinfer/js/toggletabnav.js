function toggleInputType(inputType) {
  var atlasRegionDiv = document.getElementById("atlasRegions");
  var coordsDiv = document.getElementById("coords");
  var uploadDiv = document.getElementById("upload");
  var areanav = document.getElementById("mask-input");
  var coordnav = document.getElementById("coord-input");
  var uploadnav = document.getElementById("upload-input");
  if (inputType == "mask") {
    atlasRegionDiv.style.display = "";
    coordsDiv.style.display = "none";
    uploadDiv.style.display = "none";
    coordnav.style.backgroundColor = "grey";
    areanav.style.backgroundColor = "#f3f3f3";
    uploadnav.style.backgroundColor = "grey";
  } else if (inputType == "coord") {
    atlasRegionDiv.style.display = "none";
    uploadDiv.style.display = "none";
    coordsDiv.style.display = "";
    areanav.style.backgroundColor = "grey";
    coordnav.style.backgroundColor = "#f3f3f3";
    uploadnav.style.backgroundColor = "grey";
  } else if (inputType == "upload") {
    atlasRegionDiv.style.display = "none";
    uploadDiv.style.display = "";
    coordsDiv.style.display = "none";
    areanav.style.backgroundColor = "grey";
    coordnav.style.backgroundColor = "grey";
    uploadnav.style.backgroundColor = "#f3f3f3";
  }
}
