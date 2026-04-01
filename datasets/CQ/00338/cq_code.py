import cadquery as cq

result = (
cq.Workplane("XY")
.circle(50).extrude(5)
.faces(">Z").workplane()
.sketch()
.parray(50, 0, 60, 6, rotate=False)
.circle(5)
.finalize()
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')