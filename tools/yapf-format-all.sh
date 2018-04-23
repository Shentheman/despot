# https://github.com/personalrobotics/pr_docs/wiki/Python
find "$1" -iname "*.py" -exec yapf -i {} \;
