let chart;

async function analyzeReviews() {
  const text = document.getElementById("reviewText").value.trim();
  if (!text) {
    alert("Please enter at least one review.");
    return;
  }

  const reviews = text.split("\n").filter(r => r.trim());

  let fakeCount = 0;
  let genuineCount = 0;

  const reviewList = document.getElementById("reviewList");
  reviewList.innerHTML = "";

  try {
    for (let review of reviews) {

      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded"
        },
        body: new URLSearchParams({
          review: review
        })
      });

      const data = await response.json();

      if (data.error) {
        alert(data.error);
        return;
      }

      const div = document.createElement("div");
      div.classList.add("review-item");

      if (data.result === "Fake") {
        div.classList.add("fake");
        fakeCount++;
      } else {
        div.classList.add("genuine");
        genuineCount++;
      }

      div.innerHTML = `
        <strong>${data.result}</strong>
        (Confidence: ${data.confidence}%)
        <p>${review}</p>
      `;

      reviewList.appendChild(div);
    }

    document.getElementById("totalCount").innerText = reviews.length;
    document.getElementById("fakeCount").innerText = fakeCount;
    document.getElementById("genuineCount").innerText = genuineCount;

    showResults(fakeCount, genuineCount);

  } catch (error) {
    console.error(error);
    alert("Something went wrong while analyzing.");
  }
}

function showResults(fake, genuine) {
  document.getElementById("inputView").classList.add("hidden");
  document.getElementById("resultsView").classList.remove("hidden");

  const ctx = document.getElementById("pieChart").getContext("2d");

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "pie",
    data: {
      labels: ["Fake", "Genuine"],
      datasets: [{
        data: [fake, genuine],
        backgroundColor: ["#ef4444", "#22c55e"]
      }]
    }
  });
}

function goBack() {
  document.getElementById("resultsView").classList.add("hidden");
  document.getElementById("inputView").classList.remove("hidden");
}

function logout() {
  window.location.href = "/logout";
}

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();

  reader.onload = function(e) {
    document.getElementById("reviewText").value = e.target.result;
  };

  reader.readAsText(file);
}