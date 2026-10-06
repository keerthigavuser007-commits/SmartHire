from pathlib import Path
from pypdf import PdfReader
from docx import Document


def extract_from_pdf(file_path):
    text = []

    reader = PdfReader(file_path)

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text)


def extract_from_docx(file_path):
    document = Document(file_path)

    text = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text.append(paragraph.text)

    return "\n".join(text)


def extract_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def extract_resume_text(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Resume file not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_from_pdf(file_path)

    elif extension == ".docx":
        return extract_from_docx(file_path)

    elif extension == ".txt":
        return extract_from_txt(file_path)

    else:
        raise ValueError(
            "Unsupported file type. Please use PDF, DOCX, or TXT."
        )


if __name__ == "__main__":

    print("=" * 60)
    print("SMART HIRE - RESUME PARSER")
    print("=" * 60)

    resume_path = input("\nEnter the path of your resume: ").strip()

    # Remove quotes if Windows Copy as Path adds them
    resume_path = resume_path.strip('"')

    try:

        resume_text = extract_resume_text(resume_path)

        print("\nResume extracted successfully!")

        print("\n" + "=" * 60)
        print("EXTRACTED RESUME TEXT")
        print("=" * 60)

        print(resume_text[:5000])

        print("\n" + "=" * 60)
        print("Total characters extracted:", len(resume_text))
        print("=" * 60)

    except Exception as error:

        print("\nERROR:")
        print(error)