const keycloak = new Keycloak({
            url: 'http://localhost:8080/auth',
            realm: 'myRealm',
            clientId: 'my-client'
        });

        document.getElementById('register-btn').addEventListener('click', function () {
            keycloak.register();
        });
