let mode = "obstacle";

let start = null;
let goal = null;

let obstacles = [];

const gridContainer =
    document.getElementById("grid");

function createGrid() {

    const size = parseInt(
        document.getElementById("gridSize").value
    );

    gridContainer.innerHTML = "";

    gridContainer.style.gridTemplateColumns =
        `repeat(${size}, 30px)`;

    start = null;
    goal = null;
    obstacles = [];

    for (let r = 0; r < size; r++) {

        for (let c = 0; c < size; c++) {

            const cell =
                document.createElement("div");

            cell.classList.add("cell");

            cell.dataset.row = r;
            cell.dataset.col = c;

            cell.addEventListener(
                "click",
                handleCellClick
            );

            gridContainer.appendChild(cell);
        }
    }
}

function handleCellClick(e) {

    const cell = e.target;

    const row =
        parseInt(cell.dataset.row);

    const col =
        parseInt(cell.dataset.col);

    if (mode === "start") {

        document
            .querySelectorAll(".start")
            .forEach(x =>
                x.classList.remove("start")
            );

        start = [row, col];

        cell.classList.add("start");
    }

    else if (mode === "goal") {

        document
            .querySelectorAll(".goal")
            .forEach(x =>
                x.classList.remove("goal")
            );

        goal = [row, col];

        cell.classList.add("goal");
    }

    else {

        if (
            cell.classList.contains("start") ||
            cell.classList.contains("goal")
        ) {
            return;
        }

        cell.classList.toggle("obstacle");

        if (
            cell.classList.contains("obstacle")
        ) {

            obstacles.push([row, col]);

        } else {

            obstacles =
                obstacles.filter(
                    x =>
                        !(
                            x[0] === row &&
                            x[1] === col
                        )
                );
        }
    }
}

document
    .getElementById("startBtn")
    .onclick = () => {

        mode = "start";
    };

document
    .getElementById("goalBtn")
    .onclick = () => {

        mode = "goal";
    };

document
    .getElementById("obstacleBtn")
    .onclick = () => {

        mode = "obstacle";
    };

document
    .getElementById("clearBtn")
    .onclick = () => {

        createGrid();

        document
            .getElementById("rlSteps")
            .innerText = "";

        document
            .getElementById("dijkstraSteps")
            .innerText = "";
    };

document
    .getElementById("gridSize")
    .addEventListener(
        "change",
        createGrid
    );

document
    .getElementById("runBtn")
    .onclick = async () => {

        if (!start || !goal) {

            alert(
                "Please select Start and Goal."
            );

            return;
        }

        const response =
            await fetch(
                "/simulate",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        start,
                        goal,
                        obstacles
                    })
                }
            );

        const data =
            await response.json();

        document
            .getElementById("rlSteps")
            .innerText =
            `RL Steps: ${data.rl_steps}`;

        document
            .getElementById("dijkstraSteps")
            .innerText =
            `Dijkstra Steps: ${data.dijkstra_steps}`;

        drawPath(
            data.dijkstra_path,
            "dijkstra-path"
        );

        drawPath(
            data.rl_path,
            "rl-path"
        );
    };

function drawPath(path, cls) {

    path.forEach(pos => {

        const r = pos[0];
        const c = pos[1];

        const cell =
            document.querySelector(
                `[data-row="${r}"][data-col="${c}"]`
            );

        if (
            cell &&
            !cell.classList.contains("start") &&
            !cell.classList.contains("goal")
        ) {

            cell.classList.add(cls);
        }
    });
}

createGrid();
