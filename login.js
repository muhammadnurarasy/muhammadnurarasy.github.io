
const keycloak = new Keycloak({
    url: 'http://localhost:8080/auth',
    realm: 'myRealm',
    clientId: 'my-client'
});

document.getElementById('login-btn').addEventListener('click', function () {
    keycloak.login();
});

