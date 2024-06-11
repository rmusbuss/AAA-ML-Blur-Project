const image = document.getElementById("processed-image");

const openModal = () => {
    const modal = document.getElementById("modal");
    modal.style.display = "block";
    const body = document.getElementsByTagName("body")[0];
    body.style.overflow = "hidden";
}

const closeModal = () => {
    const modal = document.getElementById("modal");
    modal.style.display = "none";
    const body = document.getElementsByTagName("body")[0];
    body.style.overflow = "auto";
}

document.addEventListener('keyup',function(e){
    if (e.key === "Escape") {
        // hide modal code
        closeModal()
    }
});

image.addEventListener("click", openModal)