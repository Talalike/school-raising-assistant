# interactive_assistant.py

from pipeline.rag_pipeline import run_pipeline

def get_user_input(prompt_text):
    print("\n" + prompt_text)
    return input("→ ")

def main():
    print("👋 Welcome to Alba – Your School Campaign Assistant!\n")

    school_name = get_user_input("🏫 Enter the name of your school:")
    project_category = get_user_input("🎨 Enter the project category:")
    user_input_1 = get_user_input("📌 What is your project about?")
    user_input_2 = get_user_input("💡 Why is this project important?")
    user_input_3 = get_user_input("🎁 Would you like to offer rewards? If yes, describe them briefly:")

    print("\n🚀 Generating your campaign draft. Please wait...\n")

    result = run_pipeline(
        school_name=school_name,
        project_category=project_category,
        user_input_1=user_input_1,
        user_input_2=user_input_2,
        user_input_3=user_input_3,
        k=5
    )
    print(f"\n📡 DEBUG – type(result): {type(result)}")

    import json # comment this line if you don't want JSON output
    print("\n🧾 Final structured output:\n") #  comment this line if you don't want JSON output
    print(json.dumps(result, indent=2, ensure_ascii=False)) #  comment this line if you don't want JSON output

if __name__ == "__main__":
    main()
