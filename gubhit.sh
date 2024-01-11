#!/bin/bash

git add .

echo -e "\033[0;32mEnter your commit message:\033[0m"
read commit_message
git commit -m "$commit_message"

echo -e "\033[0;34mPushing to GitHub...\033[0m"
git push origin main

echo -e "\033[0;32mDone!\033[0m"
