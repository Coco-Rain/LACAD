import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 10, 3)
.faces(">Z")
.workplane()
.center(-5, 0)
.slot2D(12, 4)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')