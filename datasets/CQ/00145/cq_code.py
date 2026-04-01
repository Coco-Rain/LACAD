import cadquery as cq

result = (
cq.Workplane("XY")
.polygon(6, 25)
.extrude(8)
.faces(">Z")
.workplane()
.circle(12)
.extrude(4)
.faces(">X")
.workplane()
.center(0, -6)
.slot2D(20, 5, 0)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')