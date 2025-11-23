const API_BASE = "";

async function getCF() {
    const userId = document.getElementById("cf-user-id").value;
    const res = await fetch(`${API_BASE}/recommend/${userId}`);
    const data = await res.json();
    const container = document.getElementById("cf-results");
    container.innerHTML = "";
    data.recommendations.forEach(m => {
        const card = document.createElement("div");
        card.className = "movie-card";
        card.innerHTML = `<div>${m.title}</div><div class="score">Score: ${m.score.toFixed(2)}</div>`;
        container.appendChild(card);
    });
}

async function getCB() {
    const title = document.getElementById("cb-movie-title").value;
    const res = await fetch(`${API_BASE}/recommend/content/?title=${encodeURIComponent(title)}`);
    const data = await res.json();
    const container = document.getElementById("cb-results");
    container.innerHTML = "";
    data.recommendations.forEach(m => {
        const card = document.createElement("div");
        card.className = "movie-card";
        card.innerHTML = `<div>${m.title}</div><div class="score">Score: ${m.score.toFixed(2)}</div>`;
        container.appendChild(card);
    });
}
