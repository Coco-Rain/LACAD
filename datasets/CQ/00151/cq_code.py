import cadquery as cq

result = (
cq.Workplane("XY")
.rect(20, 20)
.extrude(5)
.faces(">Z")
.workplane()
.sketch()
.push([(0, 8), (8, 0), (0, -8), (-8, 0)])
.circle(2)
.finalize()
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')