document.addEventListener("DOMContentLoaded", function() {
    const loginForm = document.getElementById('login-form');
    const registerLink = document.getElementById('register-link');

    loginForm.addEventListener('submit', function(event) {
        event.preventDefault();

        const usernameInput = document.getElementById('username');//.value;
        const passwordInput = document.getElementById('password'); //.value;
        const username = usernameInput.value;
        const password = passwordInput.value;
        // Add your authentication logic here
        if (username === 'admin' && password === '123') {
            alert('Login successful!');
            // openNewWindow('index.html');
            window.location.href = 'index.html';
            resetInputs(usernameInput, passwordInput);
            // Redirect or perform other actions after successful login
        } else {
            alert('Invalid username or password. Please try again.');
            // You can also update the UI to show an error message
            resetInputs(usernameInput, passwordInput);
        }
    });
});
function resetInputs(...inputs) {
    document.getElementById('username').value = '';
    document.getElementById('password').value = '';
}
