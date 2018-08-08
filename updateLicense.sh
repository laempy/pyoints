#!/bin/bash


LICENSE_NOTES_FILE='./templates/LICENSE_NOTES.txt'
LICENSE_FILE='./templates/GPL_LICENSE.txt'

OPENING_PATTERN='# BEGIN OF LICENSE NOTE'
CLOSING_PATTERN='# END OF LICENSE NOTE'

USAGE="Usage to update|insert|remove license: updateLicense.sh -u|-i|-r"

FILE_PATH='pointspy'
LICENSE='LICENSE'

files=$(find $FILE_PATH -type f -name "*.py")


function get_opening_pattern_line(){
    file=$1
    echo $(grep -m 1 -n "$OPENING_PATTERN" $file | grep -Eo '^[^:]+')
}

function get_closing_pattern_line(){
    file=$1
    echo $(grep -m 1 -n "$CLOSING_PATTERN" $file | grep -Eo '^[^:]+')
}


function update_license_notes(){
    file=$1
    outfile=$2
    
    echo "update license notes $outfile"
    
    begin_line=$(get_opening_pattern_line $file)
    if [ "$begin_line" = "" ]; then
      echo "opening pattern not found"
      exit 1
    fi
    
    end_line=$(get_closing_pattern_line $file)
    if [ "$end_line" = "" ]; then
      echo "closing pattern not found"
      exit 1
    fi

    local IFSr
    heading_lines=$(head -n ${begin_line} < $file)
    tailing_lines=$(sed -n ${end_line},'$p' < $file)

    echo $heading_lines > $outfile
    while read line; do
        echo "# $line" >> $outfile
    done <$LICENSE_NOTES_FILE
    echo $tailing_lines >> $outfile
}

function insert_license_notes(){
    file=$1
    outfile=$2
    
    echo "insert license notes $outfile"
    
    local IFS=
    text=$(cat $file)
    echo "$OPENING_PATTERN" > $outfile
    echo "$CLOSING_PATTERN" >> $outfile
    echo "$text" >> $outfile

    update_license_notes $outfile $outfile
}

function remove_license_notes(){
    file=$1
    outfile=$2
    
    echo "remove license notes $outfile"
    
    begin_line=$(get_opening_pattern_line $file)
    end_line=$(get_closing_pattern_line $file)
    
    if [ "$begin_line" = "" ] || [ "$end_line" = "" ]; then
      echo "pattern not found"
    else
        let "begin_line=$begin_line-1"
        let "end_line=$end_line+2"
        
        echo $begin_line
        echo $end_line
    
        local IFS=
        
        heading_lines=$(head -n ${begin_line} < $file)
        tailing_lines=$(sed -n ${end_line},'$p' < $file)

        > $outfile
        if [ ! $begin_line = "-1" ]; then
            echo "intert header"
            echo $heading_lines >> $outfile
        fi
        if [ ! $tailing_lines = "" ]; then
             echo $tailing_lines >> $outfile
        fi
       
    fi 
    
}



if [[ $1 == "" ]]; then
    echo $USAGE
    exit 0
fi
while getopts uir option; do
    case "${option}" in
    u)
        # update license notes
        echo 'update license'
        
        cp $LICENSE_NOTES_FILE $LICENSE
        echo >> $LICENSE
        echo >> $LICENSE
        echo "$(cat $LICENSE_FILE)" >> $LICENSE
               
        for file in $files; do
            update_license_notes $file $file
            exit 0
        done
        exit 0
        ;;
    i)
        echo 'insert license note'
        for file in $files; do
            insert_license_notes $file $file
            exit 0
        done
        exit 0
        ;;
    r)
        echo 'remove license note'
        for file in $files; do
            remove_license_notes $file $file
            exit 0
        done
        exit 0
        ;;
    *)
        echo $USAGE
        exit 0
        ;;
    esac
done




