import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 20, 12)
.faces(">Z")
.workplane()
.ellipse(16, 8)
.extrude(6)
.faces(">X")
.workplane()
.slot2D(18, 4)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')