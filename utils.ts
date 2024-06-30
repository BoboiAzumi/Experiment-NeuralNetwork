export function clrscr(){
    let lines = process.stdout.getWindowSize()[1];
    for(var i = 0; i < lines; i++) {
        process.stdout.write('\x1Bc')
    }
}