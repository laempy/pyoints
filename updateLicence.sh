#!/bin/bash

IFS=
LICENSE_NOTES_FILE='./templates/LICENSE_NOTES.txt'
LICENSE_FILE='./templates/GPL_LICENSE.txt'

begin_pattern='BEGIN OF LICENSE NOTE'
end_pattern='END OF LICENSE NOTE'

file='LICENSE'
outfile='UPDATED_LICENSE'

begin_line=$(grep -m 1 -n $begin_pattern $file | grep -Eo '^[^:]+')
end_line=$(grep -m 1 -n $end_pattern $file | grep -Eo '^[^:]+')

if [ "$begin_line" = "" ]
then
  echo "nope"
fi

echo $begin_line

sed -n "1p;${begin_line}p" < $file > $outfile
echo >> $outfile
cat $LICENSE_NOTES_FILE >> $outfile
echo >> $outfile
sed -n ${end_line},'$p' < $file >> $outfile


