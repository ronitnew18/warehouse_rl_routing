let mode = "obstacle";
let start = null;
let goal = null;
let obstacles = [];

const gridContainer = document.getElementById("grid");

function createGrid() {
    const size = parseInt(document.getElementById("gridSize").value);
    gridContainer.innerHTML = "";
    gridContainer.style.gridTemplateColumns = `repeat(${size}, 30px)`;

    start = null;
    goal = null;
    obstacles = [];

    for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
            const cell = document.createElement("div");
            cell.classList.add("cell");
            cell.dataset.row = r;
            cell.dataset.col = c;
            cell.addEventListener("click", handleCellClick);
            gridContainer.appendChild(cell);
        }
    }
}

function handleCellClick(e) {
    const cell = e.target;
    const row = parseInt(cell.dataset.row);
    const col = parseInt(cell.dataset.col);

    if (mode === "start") {
        document.querySelectorAll(".start").forEach(x => x.classList.remove("start"));
        start = [row, col];
        cell.classList.add("start");
    } else if (mode === "goal") {
        document.querySelectorAll(".goal").forEach(x => x.classList.remove("goal"));
        goal = [row, col];
        cell.classList.add("goal");
    } else {
        if (cell.classList.contains("start") || cell.classList.contains("goal")) return;
        cell.classList.toggle("obstacle");

        if (cell.classList.contains("obstacle")) {
            obstacles.push([row, col]);
        } else {
            obstacles = obstacles.filter(x => !(x[0] === row && x[1] === col));
        }
    }
}

// Bind event listeners natively to the IDs
document.getElementById("startBtn").onclick = () => { mode = "start"; };
document.getElementById("goalBtn").onclick = () => { mode = "goal"; };
document.getElementById("obstacleBtn").onclick = () => { mode = "obstacle"; };

document.getElementById("clearBtn").onclick = () => {
    createGrid();
    document.getElementById("rlSteps").innerText = "";
    document.getElementById("dijkstraSteps").innerText = "";
};

document.getElementById("gridSize").addEventListener("change", createGrid);

document.getElementById("runBtn").onclick = async () => {
    if (!start || !goal) {
        alert("Please select both a Start and Goal position first.");
        return;
    }

    document.getElementById("rlSteps").innerText = "Calculating paths...";
    document.getElementById("dijkstraSteps").innerText = "";

    try {
        const response = await fetch("/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ start, goal, obstacles })
        });

        const data = await response.json();

        // Display results cleanly
        document.getElementById("rlSteps").innerText = `RL Agent Path Steps: ${data.rl_steps}`;
        document.getElementById("dijkstraSteps").innerText = `Dijkstra Path Steps: ${data.dijkstra_steps}`;

        // Clear out old path colors if running multiple times
        document.querySelectorAll(".cell").forEach(cell => {
            cell.classList.remove("dijkstra-path", "rl-path");
        });

        // Draw paths on the layout
        if (data.dijkstra_path) drawPath(data.dijkstra_path, "dijkstra-path");
        if (data.rl_path) drawPath(data.rl_path, "rl-path");

    } catch (err) {
        console.error(err);
        document.getElementById("rlSteps").innerText = "Error reaching calculation server.";
    }
};

function drawPath(path, cls) {
    path.forEach(pos => {
        const r = pos[0];
        const c = pos[1];
        const cell = document.querySelector(`[data-row="${r}"][data-col="${c}"]`);
        if (cell && !cell.classList.contains("start") && !cell.classList.contains("goal")) {
            cell.classList.add(cls);
        }
    });
}

// Instantiate on startup
createGrid();
