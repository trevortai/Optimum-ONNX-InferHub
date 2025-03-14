# Description: This script is for the user to select what model to run


print("\nWhich model would you like to run?")
print("Supported Models: \n 1. Automatic Speech Recognition (Whisper) \n 2. Text Classification (Bert)")

while True:
    try:
        model = int(input("Enter your selection: "))
        if model == 1:
            from whisper.read import main
            main()
            break
        elif model == 2:
            from bert.txtclass import main
            main()
            break
        else:
            print("Please enter 1 or 2")
    except ValueError:
        print("Please enter a valid number")
        continue
    