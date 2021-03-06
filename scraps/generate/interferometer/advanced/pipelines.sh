echo "Setting up Environment variables."
export PYAUTO_PATH=/mnt/c/Users/Jammy/Code/PyAuto
export WORKSPACE_PATH=$PYAUTO_PATH/autolens_workspace

export SCRIPTS_PATH=$WORKSPACE_PATH/scripts/interferometer/advanced/pipelines
export NOTEBOOKS_PATH=$WORKSPACE_PATH/notebooks/interferometer/advanced/pipelines

echo "Removing old notebooks."
rm $NOTEBOOKS_PATH/*.ipynb
rm -rf $NOTEBOOKS_PATH/.ipynb_checkpoints

echo "Converting scripts to notebooks."
cd $SCRIPTS_PATH
py_to_notebook
cd $WORKSPACE_PATH/generate

echo "Moving Notebooks to notebooks folder."
mv $SCRIPTS_PATH/*.ipynb $NOTEBOOKS_PATH
git add $NOTEBOOKS_PATH/*.ipynb
rm $NOTEBOOKS_PATH/__init__.ipynb

echo "CLeaning up pipelines folder"
rm $SCRIPTS_PATH/*.ipynb
rm $SCRIPTS_PATH/pipelines/*.ipynb
rm $NOTEBOOKS_PATH/*.py
cp -r $SCRIPTS_PATH/pipelines $NOTEBOOKS_PATH
rm $NOTEBOOKS_PATH/__init__.ipynb
git add $NOTEBOOKS_PATH/*

echo "Renaming notebook methods"
find $NOTEBOOKS_PATH/*.ipynb -type f -exec sed -i 's/# %matplotlib/%matplotlib/g' {} +
find $NOTEBOOKS_PATH/*.ipynb -type f -exec sed -i 's/# from pyproj/from pyproj/g' {} +
find $NOTEBOOKS_PATH/*.ipynb -type f -exec sed -i 's/# workspace_path/workspace_path/g' {} +
find $NOTEBOOKS_PATH/*.ipynb -type f -exec sed -i 's/# %cd/%cd/g' {} +
find $NOTEBOOKS_PATH/*.ipynb -type f -exec sed -i 's/# print(f/print(f/g' {} +

echo "Running Notebooks"
mv $WORKSPACE_PATH/config $WORKSPACE_PATH/config_temp
mv $WORKSPACE_PATH/config_test $WORKSPACE_PATH/config
cd $NOTEBOOKS_PATH
for f in $NOTEBOOKS_PATH/*.ipynb; do jupyter nbconvert --to notebook --clear-output --execute --output ""$f"" "$f"; done
for f in $NOTEBOOKS_PATH/*.ipynb; do jupyter nbconvert --to notebook --clear-output "$f"; done
cd $WORKSPACE_PATH/generate
mv $WORKSPACE_PATH/config $WORKSPACE_PATH/config_test
mv $WORKSPACE_PATH/config_temp $WORKSPACE_PATH/config
