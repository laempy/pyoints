#!/bin/bash
LICENSE_NOTES_FILE='./templates/LICENSE_NOTES.txt'
LICENSE_FILE='./templates/GPL_LICENSE.txt'

OPENING_PATTERN='# BEGIN OF LICENSE NOTE'
CLOSING_PATTERN='# END OF LICENSE NOTE'

FILE_PATHS='pointspy examples tests'
LICENSE='LICENSE'

USAGE="Usage to update|insert|remove license: updateLicense.sh -u|-i|-r"


FILES=$(find $FILE_PATHS -type f -name "*.py")


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
    if [ ! -f $file ]; then
        echo "file $file does not exist"
        exit 1
    fi
        
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

    local IFS=
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
    if [ ! -f "$file" ]; then
        echo "file $file does not exist"
        exit 1
    fi
        
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
    if [ ! -f $file ]; then
        echo "file $file does not exist"
        exit 1
    fi

    echo "remove license notes $outfile"
    
    begin_line=$(get_opening_pattern_line $file)
    end_line=$(get_closing_pattern_line $file)
    
    if [ "$begin_line" = "" ] || [ "$end_line" = "" ]; then
        echo "nothing to remove"
    else
        let "begin_line=$begin_line-1"
        let "end_line=$end_line+1"
        
        local IFS=
        
        heading_lines=$(head -n ${begin_line} < $file)
        tailing_lines=$(sed -n ${end_line},'$p' < $file)

        > $outfile
        if [ ! $begin_line = "0" ]; then
            echo $heading_lines >> $outfile
        fi
        if [ ! $tailing_lines = "" ]; then
             echo $tailing_lines >> $outfile
        fi
       
    fi 
    
}

function create_license_file(){
    file=$1    
    cp $LICENSE_NOTES_FILE $file
    echo >> $file
    echo >> $file
    echo "$(cat $LICENSE_FILE)" >> $file
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
        
        create_license_file $LICENSE
               
        for file in $FILES; do
            update_license_notes $file $file
        done
        exit 0
        ;;
    i)
        create_license_file $LICENSE
    
        echo 'insert license note'
        for file in $FILES; do
            insert_license_notes $file $file
        done
        exit 0
        ;;
    r)
        echo 'remove license note'
        
        if [ -f $LICENSE ]; then
            rm $LICENSE
        fi
        
        for file in $FILES; do
            remove_license_notes $file $file
        done
        exit 0
        ;;
    *)
        echo $USAGE
        exit 0
        ;;
    esac
done




