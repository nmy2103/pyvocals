// Function to style the function name and parentheses
function styleFunction() {
    const textElements = document.querySelectorAll('.function-text');
    textElements.forEach(function (textElement) {
        const text = textElement.innerText;

        const regex = /([a-zA-Z0-9_]+)\((.*)\)/;

        const styledText = text.replace(regex, (match, functionName, params) => {
            return `<span class = "function-name">${functionName}</span><span class = "parentheses">(</span><span class = "params">${params}</span><span class = "parentheses">)</span>`;
        });

        textElement.innerHTML = styledText;
    });
}

document.addEventListener('DOMContentLoaded', styleFunction);