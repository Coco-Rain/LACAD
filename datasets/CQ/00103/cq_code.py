import cadquery as cq

result = (
cq.Workplane("XY")
.rect(20, 10)
.extrude(5)
.faces(">Z")
.workplane()
.slot2D(8, 2)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')