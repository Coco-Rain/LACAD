import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 30, 3)
.faces(">Z")
.workplane()
.center(-5, -5)
.circle(2)
.cutThruAll()
cq.exporters.export(result, 'GT.stl')