function previewImage(event) {
    var preview = document.getElementById('preview');
    preview.style.display = "block";
    preview.src = URL.createObjectURL(event.target.files[0]);
}
