#RTpotter v.1.0
#copyright 2016-2018 Lukasz J. Nowak
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>

# Define input parameters: input dicom filename, name of the contour sequence to be converted
# and name of the output STL file
InputFileName = "example.dcm"
OutputFileName = 'D:\DICOM\RTpotter_examples\example_avoidance0p5.stl'
CountourSequenceName ='Avoidance 0.5'










#start with importing pydicom library - it will be used for reading specific content from the DICOM file
import pydicom as dicom






#Class definitions:
#*********************************
#Class of points:
class point:
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z
	#Two points are considered equal, if all their coordinates are equal:
    def __eq__(self,other):
        if self.x == other.x and self.y == other.y and self.z == other.z:
            return True
        else:
            return False


#*********************************
#stlfacet class defines structures acordingly to the STL format. 
#Triangle elements with vertices numbered in such a way, that the normal versor points outwards    
class stlfacet:
    def __init__(self,pointa,pointb,pointc):
        self.pointa = pointa
        self.pointb = pointb
        self.pointc = pointc     
        #coefficients of vector normal to the facet surface are computed using point class:
        vectu = point(pointb.x-pointa.x,pointb.y-pointa.y,pointb.z-pointa.z)
        vectv = point(pointc.x-pointa.x,pointc.y-pointa.y,pointc.z-pointa.z)
        nnx = vectu.y*vectv.z - vectu.z*vectv.y
        nny = vectu.z*vectv.x - vectu.x*vectv.z
        nnz = vectu.x*vectv.y - vectu.y*vectv.x
        #normalization:
        self.nx = nnx / ((nnx**2 + nny**2 + nnz**2)**(1/2))
        self.ny = nny / ((nnx**2 + nny**2 + nnz**2)**(1/2))
        self.nz = nnz / ((nnx**2 + nny**2 + nnz**2)**(1/2))

    #return text with computed coordinates accordingly to the STL file standard:    
    def printfacet(self):
        return 'facet normal ' + str(self.nx) + ' ' + str(self.ny) + ' ' + str(self.nz) + '\n\t outer loop\n\t\t vertex ' + str(self.pointa.x) + ' ' + str(self.pointa.y) + ' ' + str(self.pointa.z) + '\n\t\t vertex ' + str(self.pointb.x) + ' ' + str(self.pointb.y) + ' ' + str(self.pointb.z) + '\n\t\t vertex ' + str(self.pointc.x) + ' ' + str(self.pointc.y) + ' ' + str(self.pointc.z) + '\n\t endloop\n endfacet\n'















#Function definitions:
#2D distance between two points:
def distance2d(point1,point2):
    return ((point2.x-point1.x)**2 + (point2.y-point1.y)**2) ** (1/2)


#*******************
#3D distance between two points:
def distance(point1,point2):
    return ((point2.x-point1.x)**2 + (point2.y-point1.y)**2 + (point2.z-point1.z)**2) ** (1/2)


#**************************
#Function findDirection, defined for three subsequent points at a given curve (with middle point having extreme coordinates among all the points at the curve) returns either True or False, depending on the curve orientation
#True - direction "left", false - direction "right" (accordingly to numeration of points)
def findDirection(pointa, pointb, pointc): #funkcja wyznacza direction krzywej na podstawie trzech punktow na wypuklym fragmencie brzegu
    if ((pointb.x - pointa.x) * (pointc.y-pointa.y) - (pointc.x - pointa.x) * (pointb.y - pointa.y)) != 0:
        return ((pointb.x - pointa.x) * (pointc.y-pointa.y) - (pointc.x - pointa.x) * (pointb.y - pointa.y)) < 0
    else:
        return None   #if points are collinear, direction cannot be determined -> value None is returned


#*******************************
#Write data to the STL file (input: filename and list of facets)
def printstl(InputFileName,facets):
    file = open(InputFileName,'w')
    file.write('solid DICOM_contour_model\n')
    for licz in range(len(facets)):
        file.write(facets[licz].printfacet())
    file.write('endsolid DICOM_contour_model\n')
    file.close()



    









def run(InputFileName, OutputFileName, CountourSequenceName):
    ##STEP 1
    # Read DICOM file:
    ds = dicom.read_file(InputFileName)

    # print(ds)
    # STEP 2: find the specified structure in the file (ROI contour sequence with a given name):
    for licz in range(len(ds.StructureSetROISequence)):
        if ds.StructureSetROISequence[licz][0x3006, 0x26].value == CountourSequenceName:
            contno = licz
            break
        elif licz == len(ds.StructureSetROISequence) - 1:
            raise NameError('There is no contour with the specified name within the specified DICOM file!')

    # STEP 3: determine how many slices (curves) are within the specified sequence:
    howManySlices = len(list(ds[0x3006, 0x39][contno][0x3006, 0x40]))

    # STEP 4: determine number of points in each slice (curve):
    howManyPoints = [0] * howManySlices
    for licz in range(howManySlices):
        howManyPoints[licz] = len(list(ds[0x3006, 0x39][contno][0x3006, 0x40][licz][
                                           0x3006, 0x50])) // 3  # divide by 3, because each point has 3 coordinates

    # STEP 5:
    # Read and save the coordinates of each point, in each slice (curve):
    points = [0] * howManySlices  # initialization, first dimension (slices)
    for licz in range(howManySlices):
        points[licz] = [0] * howManyPoints[licz]  # initialization, second dimension [slices][points]

    # save values to the initialized lists:
    for licz1 in range(howManySlices):
        for licz2 in range(howManyPoints[licz1]):
            points[licz1][licz2] = point(float(ds[0x3006, 0x39][contno][0x3006, 0x40][licz1][0x3006, 0x50][3 * licz2]),
                                         float(ds[0x3006, 0x39][contno][0x3006, 0x40][licz1][0x3006, 0x50][
                                                   3 * licz2 + 1]), float(
                    ds[0x3006, 0x39][contno][0x3006, 0x40][licz1][0x3006, 0x50][3 * licz2 + 2]))
    # Sort slices accordingle to the increasing z coordinate (slices are within XY plane):
    points.sort(key=lambda x: x[0].z)

    # STEP 6:
    # Remove redundant slices.
    # This version of software assumes, that only one curve per slice (i.e. with one, specified z value for all points belonging to the curve) is permitted.
    # If greater number of curves with the same z coordinate are detected, only one curve, with greatest number of points is selected for further processing.
    # Other curves are considered as redundant, and removed. In fact, many of such unwanted artifact appears as a result of automated segmentation and curve drawing processes.
    licz = 0
    while licz < len(points) - 1:
        if points[licz][0].z == points[licz + 1][0].z:
            if howManyPoints[licz] >= howManyPoints[licz + 1]:
                del (points[licz + 1])
                del (howManyPoints[licz + 1])
            else:
                del (points[licz])
                del (howManyPoints[licz])
            licz = -1
        licz += 1
    howManySlices = len(points)  # Number of remaining slices is updated

    # STEP 7:
    # numbers of points in each slice (curve) are re-arranged in such a way, that the starting points (indices 0) are the closes points between the subsequent slices (curves)
    # re-numeration starts at point 0, slice 0 (with the lowest z coordinate)
    pt0indices = [0] * howManySlices

    for licz1 in range(howManySlices - 1):
        minPointDistance = distance(points[licz1][0], points[licz1 + 1][0])  # temporal variable for storing distances
        for licz2 in range(howManyPoints[licz1 + 1]):
            if distance(points[licz1 + 1][licz2], points[licz1][0]) < minPointDistance:
                minPointDistance = distance(points[licz1 + 1][licz2],
                                            points[licz1][0])  # current lowest distance value - update
                pt0indices[licz1 + 1] = licz2
                # if starting point in the slice has different index than 0, numbering is re-arranged:
        if not pt0indices[licz1 + 1] == 0:
            points[licz1 + 1] = points[licz1 + 1][pt0indices[licz1 + 1]:] + points[licz1 + 1][:pt0indices[licz1 + 1]]

    # STEP 8
    # direction (i.e. points numeration relative to interior/exterior of the closed curve) of each curve is checked.  If differences between slices are found, numbering of points in non-matching curves is reversed
    # First, an extreme point is found (the point with maximum x or xy coordinates in the curve):
    max_xy = [0] * howManySlices
    max_yi = [0] * howManySlices
    maxIndex = [0] * howManySlices
    # point with maximum x coordinate is found:
    for licz1 in range(howManySlices):
        max_xy[licz1] = points[licz1][0].x
        max_yi[licz1] = points[licz1][0].y
        for licz2 in range(1, howManyPoints[licz1]):
            if points[licz1][licz2].x > max_xy[licz1]:
                max_xy[licz1] = points[licz1][licz2].x
                max_yi[licz1] = points[licz1][licz2].y
                maxIndex[licz1] = licz2  # index of point with maximum x coordinate is stored
            elif points[licz1][licz2].x == max_xy[licz1] and points[licz1][licz2].y > max_yi[
                licz1]:  # if more points have the same max x coordinate, the point with the highest y coordinate is selected among them:
                max_xy[licz1] = points[licz1][licz2].x
                max_yi[licz1] = points[licz1][licz2].y
                maxIndex[licz1] = licz2  # index of the selected point is stored

    # Now, having the extreme point, direction of each curve can be determined ("left" or "right" - directions of all curves should match each other). Three subsequent points are required (i.e. including neighbours of the extreme point):
    sliceDirections = [0] * howManySlices
    for licz in range(howManySlices):
        if maxIndex[licz] == 0:
            pointa = points[licz][howManyPoints[
                                      licz] - 1]  # if extreme point hax index 0, then one of its neighbours is the point with maximum index value inside the curve
        else:
            pointa = points[licz][maxIndex[licz] - 1]
        pointb = points[licz][maxIndex[licz]]
        if maxIndex[licz] == howManyPoints[licz] - 1:
            pointc = points[licz][
                0]  # if extreme point hax index of maximum value, then one of its neighbours is the point with index 0
        else:
            pointc = points[licz][maxIndex[licz] + 1]
        sliceDirections[licz] = findDirection(pointa, pointb, pointc)

    # Next, the direction of each curve with relation to the first (bottommost) slice are checked. If differences are found - the numeration of points in non-matching curves is reversed
    direction = sliceDirections[0]
    for licz in range(howManySlices):
        if sliceDirections[licz] != direction:
            points[
                licz].reverse()  # after reversing, indices must be shifted, such that 0 will be 0 again, not max index value
            points[licz] = points[licz][howManyPoints[licz] - 1:] + points[licz][:howManyPoints[licz] - 1]

    # STEP 9
    # Discretization of the lateral surface of the considered 3D structure:
    sidefacets = []  # list of the elements of the discretized lateral surface

    for licz in range(howManySlices - 1):
        pl1ind = 0  # index of point within the lower slice
        pl2ind = 0  # index of point within the upper slice
        pl1indnext = 1
        pl2indnext = 1
        end1 = pl1indnext >= howManyPoints[
            licz]  # just a precausion in case if any of the slices would consist of one point only
        end2 = pl2indnext >= howManyPoints[licz + 1]

        while (not end1) or (not end2):  # ...until all points on both curves will be included:

            condition = (distance(points[licz][pl1ind], points[licz + 1][pl2indnext]) <= distance(
                points[licz + 1][pl2ind], points[licz][
                    pl1indnext]))  # if the current point from the lower slice is closer to the next point at the upper layer, than vice-versa:

            if not end2:  # only if there are still more points to link within the upper layer:
                if condition or end1:
                    if not points[licz + 1][pl2indnext] == points[licz + 1][
                        pl2ind]:  # just a precausion in case if points would be doubled (i.e. multiple points with identical coordinates)
                        sidefacets.append(stlfacet(points[licz][pl1ind], points[licz + 1][pl2indnext],
                                                   points[licz + 1][pl2ind]))  # curve orientation!
                        pl2ind = pl2indnext
                        if pl2ind == 0:  # if the we have came back to the first point of the upper layer:
                            end2 = True
                    pl2indnext += 1
                    if pl2indnext == howManyPoints[
                        licz + 1]:  # if the current index is greater than the maximum index, nex iteration will finish in 0
                        pl2indnext = 0

            if not end1:  # until the whole lower layer will be covered:
                if (not condition) or end2:
                    if not points[licz][pl1indnext] == points[licz][
                        pl1ind]:  # just a precausion in case if points would be doubled (i.e. multiple points with identical coordinates)
                        sidefacets.append(stlfacet(points[licz + 1][pl2ind], points[licz][pl1ind],
                                                   points[licz][pl1indnext]))  # curve orientation!
                        pl1ind = pl1indnext
                        if pl1ind == 0:
                            end1 = True
                    pl1indnext += 1
                    if pl1indnext == howManyPoints[licz]:
                        pl1indnext = 0

    # STEP 10:
    # Discretization of lower and upper base surfaces of the considered 3D structure:
    direction = sliceDirections[0]  # curve orientation as set previously
    lowerCap = []  # memory allocation for lower base
    kd = points[0]

    # duplicated points are removed:
    for pointkd in kd:
        howManykd = kd.count(pointkd)
        if howManykd > 1:
            for licz in range(howManykd - 1):
                kd.remove(pointkd)

    licz = 0  # counter of subsequent points within the curve
    kdmark = True  # flag preventing curve twisting
    directiond = direction
    while len(kd) >= 3:  # until at least 3 points are left within the curve...
        if findDirection(kd[licz], kd[licz + 1], kd[
            licz + 2]) == directiond:  # if we are at the convex part of curve, we create facet and remove the middle point from further considerations
            if not directiond:  # normal versor to the lower base surface must point downwards
                lowerCap.append(stlfacet(kd[licz], kd[licz + 2], kd[licz + 1]))
            else:
                lowerCap.append(stlfacet(kd[licz], kd[licz + 1], kd[licz + 2]))
            kd.remove(kd[licz + 1])
            kdmark = True
        licz += 1
        if licz >= len(kd) - 2:
            if not kdmark:
                directiond = not directiond
            licz = 0
            kdmark = False

            # Next, we perform analogous processing for the upper base surface:
    upperCap = []
    kg = points[howManySlices - 1]

    ###duplicated points are removed:
    for pointkg in kg:
        howManykg = kg.count(pointkg)
        if howManykg > 1:
            for licz in range(howManykg - 1):
                kg.remove(pointkg)
                print('point usuniety z kapsla gornego!')

    licz = 0
    kgmark = True
    directiong = direction
    while len(kg) >= 3:  # until at least 3 points are left within the curve...
        if findDirection(kg[licz], kg[licz + 1], kg[
            licz + 2]) == directiong:  # if we are at the convex part of curve, we create facet and remove the middle point from further considerations
            if not directiong:
                upperCap.append(stlfacet(kg[licz], kg[licz + 1], kg[licz + 2]))
            else:
                upperCap.append(stlfacet(kg[licz], kg[licz + 2], kg[licz + 1]))
            kg.remove(kg[licz + 1])
            kgmark = True
        licz += 1
        if licz >= len(kg) - 2:
            if not kgmark:
                directiong = not directiong
            licz = 0
            kgmark = False

            # STEP 11:
    # We save the discretized structure into the STL file with the specified name, using the defined printstl function:
    printstl(OutputFileName, lowerCap + sidefacets + upperCap)

