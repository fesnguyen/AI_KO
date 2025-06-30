from pywinauto.application import Application

# Find the BlueStacks window
# blue_stacks_app = pywinauto.application.Application().connect(title="BlueStacks")
# blue_stacks_window = blue_stacks_app.top_window()
#
# # Simulate a key press to take a screenshot
# blue_stacks_window.send_keys("{PRTSC}")
#
# # Save the screenshot (you might need to adjust the file path)
# blue_stacks_window.send_keys("{CTRL+S}")


app = Application().connect(title='BlueStacks App Player')

blue_stacks_window = app.top_window()

# Get the top-level window of each application
print(app)