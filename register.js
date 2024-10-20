document.getElementById('register-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const username = document.getElementById('username').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    try {
        const response = await fetch('http://localhost:3000/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, email, password })
        });

        if (response.ok) {
            const data = await response.json();
            console.log('Registration successful:', data);
            alert('Registration successful');
        } else {
            alert('Registration failed');
        }
    } catch (error) {
        console.error('Error during registration:', error);
        alert('Error during registration');
    }
});
