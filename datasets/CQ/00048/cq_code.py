import cadquery as cq

result = (
cq.Workplane("XY")
.circle(15)
.extrude(2)
.faces(">Z")
.workplane()
.sketch()
.rarray(5, 0, 360, 8)
.circle(1)
.finalize()
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')