import cadquery as cq

result = (
cq.Workplane("XY")
.circle(6)
.extrude(4)
.faces(">Z")
.workplane()
.sketch()
.circle(5, mode='a', tag='outer')
.circle(3, mode='s')
.finalize()
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')