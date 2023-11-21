import numpy as np
import sys
import lxml.etree as ET
import cv2
import time
import os

def xml_to_mask(xml_path, location, size, tree=None, downsample=1, verbose=0):

    # parse xml and get root
    if tree == None: tree = ET.parse(xml_path)
    root = tree.getroot()

    # calculate region bounds
    bounds = {'x_min' : location[0], 'y_min' : location[1], 'x_max' : location[0] + size[0]*downsample, 'y_max' : location[1] + size[1]*downsample}
    IDs = regions_in_mask(xml_path=xml_path, root=root, tree=tree, bounds=bounds, verbose=verbose)

    if verbose != 0:
        print('\nFOUND: ' + str(len(IDs)) + ' regions')

    # find regions in bounds
    Regions = get_vertex_points(root=root, IDs=IDs, verbose=verbose)
    # print('Regions:', Regions)

    # fill regions and create mask
    mask = Regions_to_mask(Regions=Regions, bounds=bounds, IDs=IDs, downsample=downsample, verbose=verbose)
    if verbose != 0:
        print('done...\n')

    return mask

def xml_to_masks(xml_path, location, size, tree=None, downsample=1, verbose=0, linecolors=[]):
    # parse xml and get root
    if tree == None: tree = ET.parse(xml_path)
    root = tree.getroot()

    # calculate region bounds
    bounds = {
        'x_min' : location[0], 
        'y_min' : location[1], 
        'x_max' : location[0] + size[0]*downsample, 
        'y_max' : location[1] + size[1]*downsample}
    # bounds_array = np.array([bounds['x_min'], bounds['y_min'], bounds['x_max'], bounds['y_max']])

    IDs = labels_in_mask(xml_path=xml_path, root=root, tree=tree, bounds=bounds, verbose=verbose, linecolors=linecolors)

    if verbose != 0:
        print('\nFOUND: ' + str(len(IDs)) + ' regions')

    # find regions in bounds
    # print('Regions:', Regions)

    # fill regions and create mask
    masks = []
    for id in IDs:
        Regions = get_vertex_points(root=root, IDs=[id], verbose=verbose)
        for Region in Regions:
            mask = Regions_to_mask(Regions=[Region], bounds=bounds, IDs=[id], downsample=downsample, verbose=verbose)
            masks.append(mask)

    if verbose != 0:
        print('done...\n')

    return masks

def labels_in_mask(xml_path, root, tree, bounds, verbose=1, linecolors=[]):
    # find regions to save
    colors = []
    IDs = []
    mtime = os.path.getmtime(xml_path)

    write_minmax_to_xml(xml_path, tree)

    for Annotation in root.findall("./Annotation"): # for all annotations
        annotationID = Annotation.attrib['Id']
        LineColor = Annotation.attrib['LineColor']
        colors.append(LineColor)
    
    colors = np.unique(colors)
    # Encode LineColor to 0,1,2,3
    if linecolors != []:
        for id in colors:
            IDs.append({'label_id' : linecolors.index(id) + 1, 'LineColor' : id})
    
    return IDs

def get_vertex_points(root, IDs, verbose=1):
    Regions = []
    for ID in IDs: # for all IDs
        # get all vertex attributes (points)
        for Region in root.findall("./Annotation[@LineColor='" + ID['LineColor'] + "']/*/Region"):
            Vertices = []
            # Ignore all regions with area 0
            if Region.attrib['Area'] != '0.0':
                for Vertex in Region.findall("./Vertices/Vertex"):
                    # make array of points
                    Vertices.append([int(float(Vertex.attrib['X'])), int(float(Vertex.attrib['Y']))])
            Regions.append(np.array(Vertices))
    return Regions

def regions_in_mask(xml_path, root, tree, bounds, verbose=1, linecolors=[]):
    # find regions to save
    IDs = []
    mtime = os.path.getmtime(xml_path)

    write_minmax_to_xml(xml_path, tree)

    for Annotation in root.findall("./Annotation"): # for all annotations
        annotationID = Annotation.attrib['Id']
        LineColor = Annotation.attrib['LineColor']
        for Region in Annotation.findall("./*/Region"): # iterate on all regions
            # Check if region's area's is more than 0 
            print('Region area:', Region.attrib['Area'])
            if Region.attrib['Area'] != '0.0':
                for Vert in Region.findall("./Vertices"): # iterate on all vertex in region
                    # get minmax points
                    Xmin = np.int32(Vert.attrib['Xmin'])
                    Ymin = np.int32(Vert.attrib['Ymin'])
                    Xmax = np.int32(Vert.attrib['Xmax'])
                    Ymax = np.int32(Vert.attrib['Ymax'])
                    # test minmax points in region bounds
                    if bounds['x_min'] <= Xmax and bounds['x_max'] >= Xmin and bounds['y_min'] <= Ymax and bounds['y_max'] >= Ymin:
                        if verbose: print('Saving region IDs...')
                        IDs.append({'annotationID' : annotationID, 'LineColor' : LineColor})
                        break
    
    # Encode LineColor to 0,1,2,3
    if linecolors != []:
        for id in IDs:
            id['LineColor'] = linecolors.index(id['LineColor']) + 1
    
    return IDs

def Regions_to_mask(Regions, bounds, IDs, downsample, verbose=1):
    # downsample = int(np.round(downsample_factor**(.5)))

    if verbose !=0:
        print('MAKING MASK for:', IDs)

    if len(Regions) != 0: # regions present
        # get min/max sizes
        min_sizes = np.empty(shape=[2,0], dtype=np.int32)
        max_sizes = np.empty(shape=[2,0], dtype=np.int32)
        for Region in Regions: # fill all regions
            min_bounds = np.reshape((np.amin(Region, axis=0)), (2,1))
            max_bounds = np.reshape((np.amax(Region, axis=0)), (2,1))
            min_sizes = np.append(min_sizes, min_bounds, axis=1)
            max_sizes = np.append(max_sizes, max_bounds, axis=1)
        min_size = np.amin(min_sizes, axis=1)
        max_size = np.amax(max_sizes, axis=1)

        # add to old bounds
        bounds['x_min_pad'] = min(min_size[1], bounds['x_min'])
        bounds['y_min_pad'] = min(min_size[0], bounds['y_min'])
        bounds['x_max_pad'] = max(max_size[1], bounds['x_max'])
        bounds['y_max_pad'] = max(max_size[0], bounds['y_max'])

        # make blank mask
        mask = np.zeros([ int(np.round((bounds['y_max_pad'] - bounds['y_min_pad']) / downsample)), int(np.round((bounds['x_max_pad'] - bounds['x_min_pad']) / downsample)) ], dtype=np.uint8)

        # fill mask polygons
        for Region in Regions:
            # reformat Regions
            Region[:,1] = np.int32(np.round((Region[:,1] - bounds['y_min_pad']) / downsample))
            Region[:,0] = np.int32(np.round((Region[:,0] - bounds['x_min_pad']) / downsample))
            # get annotation ID for mask color
            # print('IDs inside:', IDs)
            ID = IDs[0]
            cv2.fillPoly(img=mask, pts=[Region], color=int(ID['label_id']))
        

        # reshape mask
        x_start = np.int32(np.round((bounds['x_min'] - bounds['x_min_pad']) / downsample))
        y_start = np.int32(np.round((bounds['y_min'] - bounds['y_min_pad']) / downsample))
        x_stop = np.int32(np.round((bounds['x_max'] - bounds['x_min_pad']) / downsample))
        y_stop = np.int32(np.round((bounds['y_max'] - bounds['y_min_pad']) / downsample))
        # pull center mask region
        mask = mask[ y_start:y_stop, x_start:x_stop ]

    else: # no Regions
        mask = np.zeros([ int(np.round((bounds['y_max'] - bounds['y_min']) / downsample)), int(np.round((bounds['x_max'] - bounds['x_min']) / downsample)) ])

    return mask

def write_minmax_to_xml(xml_path, tree=None, time_buffer=10):
    # function to write min and max verticies to each region

    # parse xml and get root
    if tree == None: tree = ET.parse(xml_path)
    root = tree.getroot()

    try:
        # has the xml been modified to include minmax
        modtime = np.float64(root.attrib['modtime'])
        # has the minmax modified xml been changed?
        assert os.path.getmtime(xml_path) < modtime + time_buffer

    except:

        for Annotation in root.findall("./Annotation"): # for all annotations
            annotationID = Annotation.attrib['Id']

            for Region in Annotation.findall("./*/Region"): # iterate on all region

                for Vert in Region.findall("./Vertices"): # iterate on all vertex in region
                    Xs = []
                    Ys = []
                    for Vertex in Vert.findall("./Vertex"): # iterate on all vertex in region
                        # get points
                        Xs.append(np.int32(np.float64(Vertex.attrib['X'])))
                        Ys.append(np.int32(np.float64(Vertex.attrib['Y'])))

                    # find min and max points
                    Xs = np.array(Xs)
                    Ys = np.array(Ys)

                    # modify the xml
                    Vert.set("Xmin", "{}".format(np.min(Xs)))
                    Vert.set("Xmax", "{}".format(np.max(Xs)))
                    Vert.set("Ymin", "{}".format(np.min(Ys)))
                    Vert.set("Ymax", "{}".format(np.max(Ys)))

        root.set("modtime", "{}".format(time.time()))
        xml_data = ET.tostring(tree, pretty_print=True)
        #xml_data = Annotations.toprettyxml()
        f = open(xml_path, 'w')
        f.write(xml_data.decode())
        f.close()

def get_num_classes(xml_path):
    # parse xml and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotation_num = 0
    for Annotation in root.findall("./Annotation"): # for all annotations
        annotation_num += 1

    return annotation_num + 1
