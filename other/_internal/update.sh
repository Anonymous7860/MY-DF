#!/bin/bash

cd DeepFaceLab

# Fetch changes
git fetch origin

LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse "@{u}")
BASE=$(git merge-base @ "@{u}")

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "DFL is up-to-date!"
    exit 0
elif [ "$LOCAL" = "$BASE" ]; then
    if ! git diff-index --quiet HEAD; then
        read -p "Warning: Changes to DFL detected, discard changes and update? (Y/n): " USER_INPUT

        if [[ "$USER_INPUT" =~ ^[Yy]$ || -z "$USER_INPUT" ]]; then
	    echo "Updating..."
            git reset HEAD
	    git pull origin master
    	else
            echo "Update aborted."
            exit 0
    	fi
    else
        echo "New version found, updating..."
    	git pull origin master
        exit 0
    fi
else
    read -p "Unexpected error - Do you want to force update? (This will discard any changes to DFL you made) (Y/n): " USER_INPUT

    if [[ "$USER_INPUT" =~ ^[Yy]$ || -z "$USER_INPUT" ]]; then
        echo "Updating..."
        git reset HEAD
	git pull origin master
        echo "Changes discarded."
    else
        echo "Update aborted."
        exit 0
    fi
fi

cp -f other/* ../../
cp -f other/i_internal/setenv.sh ../

./venv/bin/pip install --upgrade pip
./venv/bin/pip install --no-deps --upgrade -r requirements.txt

echo "Update finished!"