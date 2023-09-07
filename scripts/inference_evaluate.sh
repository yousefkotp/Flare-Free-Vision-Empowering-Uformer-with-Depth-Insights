input_dir=$1
output_dir=$2
model_path=$3

cmd.exe /C "delete_all_desktop_ini.bat"
python basicsr/inference.py --input=$input_dir --output="$output_dir/" --model_path=$model_path --flare7kpp
python evaluate.py --input="$output_dir/blend" --gt=dataset/Flare7Kpp/test_data/real/gt --mask=dataset/Flare7Kpp/test_data/real/mask
