from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import aiofiles
import csv
from src.helper import llm_pipeline

app = FastAPI()

# Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(request: Request, pdf_file: UploadFile = File(...), filename: str = Form(...)):
    base_folder = "static/docs"

    # 🧹 Clean if a file (not a folder) exists with the same name
    if os.path.exists(base_folder) and not os.path.isdir(base_folder):
        os.remove(base_folder)

    os.makedirs(base_folder, exist_ok=True)

    pdf_path = os.path.join(base_folder, filename)

    async with aiofiles.open(pdf_path, "wb") as f:
        content = await pdf_file.read()
        await f.write(content)

    return {"msg": "success", "pdf_filename": pdf_path}


def get_csv(file_path: str) -> str:
    answer_generation_chain, ques_list = llm_pipeline(file_path)

    base_folder = "static/output"

    # 🧹 Clean if a file (not a directory) exists
    if os.path.exists(base_folder) and not os.path.isdir(base_folder):
        os.remove(base_folder)

    os.makedirs(base_folder, exist_ok=True)
    output_file = os.path.join(base_folder, "QA.csv")

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])

        for question in ques_list:
            print("Question:", question)
            answer = answer_generation_chain.run(question)
            print("Answer:", answer)
            print("--------------------------------------------------\n")
            csv_writer.writerow([question, answer])

    return output_file


@app.post("/analyze")
async def analyze(request: Request, pdf_filename: str = Form(...)):
    output_file = get_csv(pdf_filename)
    return JSONResponse({"output_file": output_file})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
