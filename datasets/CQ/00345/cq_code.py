import cadquery as cq

result = (
cq.Workplane("XY")
.box(40, 25, 15)
.union(
cq.Workplane("XY")
.transformed(offset=(10, 0, 7.5))
.cylinder(12, 8)
)
.faces(">X")
.workplane(centerOption="CenterOfBoundBox")
.cboreHole(5, 10, 4, 8)
)
cq.exporters.export(result, 'GT.stl')