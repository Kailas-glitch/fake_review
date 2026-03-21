document.getElementById("loginForm").addEventListener("submit", function (e) {
  e.preventDefault();

  const formData = new FormData();
  formData.append("username", document.querySelector("input[name='username']").value);
  formData.append("password", document.querySelector("input[name='password']").value);

  fetch("/login", {
    method: "POST",
    body: formData
  })
  .then(response => {
    if (response.redirected) {
      window.location.href = response.url;  // ✅ Flask handles redirect
    } else {
      alert("Invalid Login");
    }
  })
  .catch(error => {
    console.error(error);
    alert("Login failed");
  });
});