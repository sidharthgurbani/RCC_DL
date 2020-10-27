stl_file_list = fopen('stl_files.txt');
image_file_list = fopen('image_files.txt');

stl_file = fgetl(stl_file_list);
image_file = fgetl(image_file_list);

while ischar(stl_file) && ischar(image_file)
    data = stlread(stl_file);
%     disp(image_file);
    tri_data = trisurf(data,'EdgeColor','k');
    saveas(tri_data, image_file);
    stl_file = fgetl(stl_file_list);
    image_file = fgetl(image_file_list);
end

fclose(stl_file_list);
fclose(image_file_list);