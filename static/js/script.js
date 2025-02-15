function startPrediction() {
    alert("Brain Tumor Prediction will start soon!");
}
// Optional: Add click event listener for debugging or analytics
document.querySelectorAll('.portfolio-link').forEach(link => {
    link.addEventListener('click', (event) => {
        console.log(`Redirecting to: ${event.target.href}`);
        // Optional: Add analytics tracking or prevent default behavior
    });
});

// Ensure Owl Carousel works (if not already included)
$(document).ready(function () {
    $(".gallery-list").owlCarousel({
        items: 3,
        margin: 20,
        loop: true,
        autoplay: true,
        autoplayTimeout: 5000,
        nav: true,
    });
});


// script.js

const images = document.querySelectorAll('.gallery img');

images.forEach((img) => {
    img.addEventListener('mouseover', () => {
        img.title = `This is ${img.alt}`;
    });
});



// Automatically submit the form after files are selected
document.addEventListener('DOMContentLoaded', function() {
    const uploadButton = document.getElementById('uploadButton');
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const submitBtn = document.getElementById('submitBtn');

    uploadButton.addEventListener('click', function() {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            fileName.textContent = `Selected file: ${this.files[0].name}`;
            submitBtn.style.display = 'inline-block';
        }
    });
});