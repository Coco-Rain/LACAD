import cadquery as cq

result = (
cq.Workplane("XY")
.polygon(6, 25)
.extrude(8)
.faces(">Z")
.workplane()
.cylinder(12, 8, combine=False)
.faces(">Z")
.workplane()
.transformed(offset=(15, 0, -5))
.cskHole(6, 10, 10, 4)
.combine()
.edges()
.fillet(2)
)
cq.exporters.export(result, 'GT.stl')