import os
import sys

from topcoffea.modules.HTMLGenerator import *
# Create an index.html file for a web-directory with .png files

STYLE_STR = """
.image {
    float:left; margin: 5px; clear:justify;
    font-size: 10px; font-family: Verdana, Arial, sans-serif;
    text-align: center;
}

.topnav {
    overflow: hidden;
    background-color: #e9e9e9;
}

.topnav a {
    float: left;
    display: block;
    color: black;
    text-align: center;
    padding: 14px 16px;
    text-decoration: none;
}

.topnav a:hover {
    background-color: #ddd;
    color: black;
}

.topnav a.active {
    background-color: #2196F3;
    color: white;
}

.topnav input[type=text] {
    float: right;
    padding: 6px;
    margin-top: 8px;
    margin-right: 16px;
    border: none;
    font-size: 17px;
}
"""

# Return list of files with a specified file_type in a directory
def getImages(tar_dir,file_type='png'):
    # Note: Will break if a filename has more then 1 . in the name
    fnames = []
    for out in sorted(os.listdir(tar_dir)):
        fpath = os.path.join(tar_dir,out)
        if (os.path.isdir(fpath)):
            continue
        f,ftype = out.rsplit(".",1)
        if ftype != file_type:
            continue
        fnames.append(out)

    return fnames

# Creates an index.html file at the specified location for displaying .png files in a web-browser
def make_html(tar_dir, width=355, height=355):
    home_dir = os.getcwd()
    if not os.path.exists(tar_dir):
        print(f"Target directory does not exists: {tar_dir}")
        return

    os.chdir(tar_dir)

    my_html = HTMLGenerator()

    meta_tag = MetaTag(); my_html.addHeadTag(meta_tag)
    meta_tag.addAttributes(charset='UTF-8')

    style_tag = StyleTag(); my_html.addHeadTag(style_tag)
    style_tag.setContent(STYLE_STR)
    style_tag.addAttributes(type='text/css')

    #topnav = DivisionTag()
    #topnav.addAttributes(cls='topnav')
    #input_tag = InputTag()
    #input_tag.addAttributes(type='text',placeholder='Search...',id='myInput')
    #my_html.addBodyTag(topnav)
    #topnav.addTag(input_tag)

    image_files = getImages(tar_dir)
    for fname in image_files:
        image_name,ftype = fname.rsplit(".",1)

        div_tag   = DivisionTag(); my_html.addBodyTag(div_tag)
        image_tag = ImgTag()
        text_div  = DivisionTag()
        if os.path.exists(fname.replace('png','pdf')):
            fname = fname.replace('png','pdf')
        link_tag  = HyperLinkTag(link_location="./%s" % (fname),link_name='')
        fname = fname.replace('pdf','png')

        # This ensures the pretty_print setting gets inherited properly
        div_tag.addTag(link_tag); div_tag.addTag(text_div)
        link_tag.addTag(image_tag)


        #image_tag.addAttributes(width=355,height=229,border=0,src="./%s" % (fname))
        image_tag.addAttributes(width=width,height=height,border=0,src="./%s" % (fname))

        link_tag.addAttributes(target='_blank')

        text_div.addAttributes(style=f'width:{width}px',id='imgName')
        text_div.setContent(image_name)

        #div_tag.addAttribute('class','image')
        div_tag.addAttributes(cls='image')


    #print my_html.dumpHTML()
    my_html.saveHTML(f_name='index.html',f_dir=tar_dir)


    os.chdir(home_dir)

def main():
    web_area = "/afs/crc.nd.edu/user/a/awightma/www/"
    if len(sys.argv) == 2:
        fpath = sys.argv[1]
    else:
        fpath = os.path.join(web_area,'eft_stuff/tmp')
    if not os.path.exists(fpath):
        print("ERROR: Unknown path {}".format(fpath))
        return
    make_html(fpath)

if __name__ == "__main__":
    main()
