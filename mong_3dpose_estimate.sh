# !/bin/bash
# i=0
n=$1

for f in openpose/sample_images/*; do
    
    # if [ $i -gt 20 ]
    # then 
    #     echo "terminated"
    #     break
    # fi
    # ((i++))
    filename=$(basename -- "$f")
  no_ext="${filename%.*}"
  tmp=$(printf "%s_keypoints.json" $no_ext)
  
  echo "Processing $no_ext"
  echo "N: $n"
  python2 hmr/mong_demo.py --img_path $f \
                     --json_path openpose/sample_jsons/$tmp \
					 --num_people $n
  
done

echo "Done"
