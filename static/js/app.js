$(document).ready(function(){
    $('.fa-bars').click(function() {
        $(this).toggleClass('fa-times');
        $('.nav').toggleClass('nav-toggle');
    });
    $(window).on('load scroll',function(){
        $('.fa-bars').removeClass('fa-times');
        $('.nav').removeClass('nav-toggle');

        if($(window).scrollTop()>10){
            $('header').addClass('header-active');
        }
        else{
            $('header').removeClass('header-active');
        }
    });


});
document.addEventListener("DOMContentLoaded", function() {
    const loadingContainer = document.getElementById('loading-container');
    const progressBar = document.getElementById('progress-bar');

    window.addEventListener('beforeunload', function() {
        loadingContainer.style.display = 'block';
        progressBar.style.width = '100%';
    });

    window.addEventListener('load', function() {
        loadingContainer.style.display = 'none';
        progressBar.style.width = '0';
    });
});

