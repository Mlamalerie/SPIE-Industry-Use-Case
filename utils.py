def display_loading_bar(i, n, bar_length=20, text=""):
    percent = float(i) / n
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    print(f"{2}: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))), 'Loading' if not text else text,
          end="\r")
